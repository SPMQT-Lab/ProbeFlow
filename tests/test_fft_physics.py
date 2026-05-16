"""Tests for FFT filter physics: fourier_filter, periodic_notch_filter,
fft_soft_border, and fft_magnitude."""

from __future__ import annotations

import math

import numpy as np
import pytest

from probeflow.processing.image import (
    fourier_filter,
    fft_magnitude,
    fft_soft_border,
    periodic_notch_filter,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _checkerboard(Ny: int = 32, Nx: int = 32) -> np.ndarray:
    """Alternating 0/1 pattern — all energy at Nyquist."""
    y, x = np.mgrid[:Ny, :Nx]
    return ((y + x) % 2).astype(np.float64)


def _uniform(Ny: int = 32, Nx: int = 32, val: float = 3.0) -> np.ndarray:
    return np.full((Ny, Nx), val, dtype=np.float64)


def _low_freq_sine(Ny: int = 32, Nx: int = 32) -> np.ndarray:
    """Single low-frequency cosine: 1 cycle across the image."""
    x = np.linspace(0, 2 * math.pi, Nx, endpoint=False)
    return np.outer(np.ones(Ny), np.cos(x))


def _cosine_at_freq(Ny: int = 64, Nx: int = 64, dx: int = 8) -> np.ndarray:
    """Cosine whose FFT peak is at ±dx bins from the DC centre."""
    freq = dx / Nx
    x = np.arange(Nx, dtype=np.float64)
    return np.outer(np.ones(Ny, dtype=np.float64), np.cos(2 * math.pi * freq * x))


# ── fourier_filter ────────────────────────────────────────────────────────────

class TestFourierFilter:
    def test_mean_approximately_preserved_high_pass(self):
        # High-pass removes high frequencies from windowed signal but adds mean back.
        # The output mean should be close to the input mean.
        rng = np.random.RandomState(42)
        arr = rng.randn(32, 32) + 5.0
        out = fourier_filter(arr, mode="high_pass", cutoff=0.3)
        assert abs(float(np.mean(out)) - float(np.mean(arr))) < 0.5

    def test_mean_approximately_preserved_low_pass(self):
        rng = np.random.RandomState(7)
        arr = rng.randn(32, 32) + 7.0
        out = fourier_filter(arr, mode="low_pass", cutoff=0.3)
        assert abs(float(np.mean(out)) - float(np.mean(arr))) < 0.5

    def test_nan_pixels_restored_after_filter(self):
        arr = np.ones((16, 16), dtype=np.float64)
        arr[5, 5] = np.nan
        arr[10, :] = np.nan
        out = fourier_filter(arr, mode="low_pass", cutoff=0.5)
        assert np.isnan(out[5, 5])
        assert np.all(np.isnan(out[10, :]))
        assert np.isfinite(out[0, 0])

    def test_low_pass_suppresses_checkerboard(self):
        # Checkerboard is at Nyquist; low-pass with small cutoff should suppress it.
        arr = _checkerboard(32, 32)
        out = fourier_filter(arr, mode="low_pass", cutoff=0.05)
        # Standard deviation of output should be much less than input
        assert float(np.std(out)) < 0.1 * float(np.std(arr))

    def test_high_pass_suppresses_uniform_image(self):
        # A uniform image has only DC; high-pass removes DC, leaving near-zero
        # variation (the mean is added back via mean_val, so abs values should
        # be close to the mean, i.e. low std).
        arr = _uniform(32, 32, val=5.0)
        out = fourier_filter(arr, mode="high_pass", cutoff=0.1)
        assert float(np.std(out)) < 1e-6

    def test_low_pass_cutoff_1_returns_input(self):
        # Early-exit path: cutoff >= 1.0 for low_pass returns filled input
        arr = np.arange(64.0, dtype=np.float64).reshape(8, 8)
        out = fourier_filter(arr, mode="low_pass", cutoff=1.0)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_high_pass_cutoff_0_returns_input(self):
        # Early-exit path: cutoff <= 0.0 for high_pass returns filled input
        arr = np.arange(64.0, dtype=np.float64).reshape(8, 8)
        out = fourier_filter(arr, mode="high_pass", cutoff=0.0)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_output_shape_preserved(self):
        arr = np.ones((20, 30))
        out = fourier_filter(arr, mode="low_pass", cutoff=0.3)
        assert out.shape == (20, 30)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            fourier_filter(np.ones((8, 8)), mode="band_pass", cutoff=0.3)

    def test_invalid_cutoff_raises(self):
        with pytest.raises(ValueError, match="cutoff"):
            fourier_filter(np.ones((8, 8)), mode="low_pass", cutoff=1.5)


# ── periodic_notch_filter ─────────────────────────────────────────────────────

class TestPeriodicNotchFilter:
    def test_suppresses_periodic_pattern(self):
        # Cosine at frequency dx=8/Nx → peak at FFT bin 8 from centre
        Ny, Nx = 64, 64
        dx = 8
        arr = _cosine_at_freq(Ny, Nx, dx=dx)
        before_std = float(np.std(arr))
        out = periodic_notch_filter(arr, peaks=[(dx, 0)], radius_px=4.0)
        after_std = float(np.std(out))
        # Pattern should be substantially suppressed
        assert after_std < 0.2 * before_std

    def test_conjugate_symmetry_both_peaks_notched(self):
        # Both (dx, dy) and (-dx, -dy) must be notched.
        # Use an image with ONLY that frequency; if only one conjugate is notched,
        # the real-valued reconstruction would still contain the pattern.
        Ny, Nx = 64, 64
        dx = 10
        arr = _cosine_at_freq(Ny, Nx, dx=dx)
        out_both = periodic_notch_filter(arr, peaks=[(dx, 0)], radius_px=4.0)
        # The single cosine with both conjugates removed leaves near-zero residual
        assert float(np.std(out_both)) < 0.2

    def test_nan_input_handled(self):
        arr = np.ones((32, 32), dtype=np.float64)
        arr[5, 5] = np.nan
        out = periodic_notch_filter(arr, peaks=[(4, 0)])
        assert np.isnan(out[5, 5])
        assert np.isfinite(out[0, 0])

    def test_empty_peaks_returns_input(self):
        arr = np.random.RandomState(0).randn(16, 16)
        out = periodic_notch_filter(arr, peaks=[])
        np.testing.assert_array_equal(out, arr.astype(np.float64))

    def test_dc_peak_skipped(self):
        # (0, 0) peak is the DC term; the implementation skips it explicitly
        arr = _low_freq_sine(32, 32)
        out = periodic_notch_filter(arr, peaks=[(0, 0)])
        # Nothing was notched, so output should equal input (approximately)
        np.testing.assert_allclose(out, arr.astype(np.float64), atol=1e-10)

    def test_output_shape_preserved(self):
        arr = np.ones((20, 30))
        out = periodic_notch_filter(arr, peaks=[(3, 2)])
        assert out.shape == (20, 30)

    def test_unaffected_frequency_not_suppressed(self):
        # Notch at dx=8 should not suppress content at dx=16
        Ny, Nx = 64, 64
        arr = _cosine_at_freq(Ny, Nx, dx=16)
        before_std = float(np.std(arr))
        out = periodic_notch_filter(arr, peaks=[(8, 0)], radius_px=2.0)
        after_std = float(np.std(out))
        # Content at dx=16 should be mostly preserved (not much suppression)
        assert after_std > 0.5 * before_std


# ── fft_soft_border ───────────────────────────────────────────────────────────

class TestFFTSoftBorder:
    def test_output_shape_preserved(self):
        arr = np.ones((24, 32))
        out = fft_soft_border(arr, mode="low_pass", cutoff=0.3)
        assert out.shape == (24, 32)

    def test_constant_image_unchanged(self):
        # centered = 0, tapered = 0, ifft = 0, then 0 / safe_win + mean_val = mean_val
        arr = _uniform(32, 32, val=4.0)
        out = fft_soft_border(arr, mode="low_pass", cutoff=0.5)
        np.testing.assert_allclose(out, arr, atol=1e-10)

    def test_low_pass_interior_close_to_input_for_smooth_image(self):
        # A low-frequency sine passes through low_pass with high cutoff;
        # the taper-recovery (/ win2d) should restore the interior.
        Ny, Nx = 64, 64
        x = np.linspace(0, 2 * math.pi, Nx, endpoint=False)
        arr = np.outer(np.ones(Ny), np.sin(x))  # freq = 1/Nx cycle/px
        out = fft_soft_border(arr, mode="low_pass", cutoff=0.9, border_frac=0.1)
        # Interior (away from taper edges): should be close to input
        interior = out[8:-8, 8:-8]
        expected = arr[8:-8, 8:-8]
        assert np.allclose(interior, expected, atol=0.05)

    def test_nan_border_runs_without_error(self):
        arr = np.ones((32, 32), dtype=np.float64)
        arr[0, :] = np.nan
        arr[-1, :] = np.nan
        arr[:, 0] = np.nan
        arr[:, -1] = np.nan
        out = fft_soft_border(arr, mode="low_pass", cutoff=0.3)
        assert out.shape == (32, 32)

    def test_nan_pixels_restored(self):
        arr = np.ones((16, 16), dtype=np.float64)
        arr[3, 3] = np.nan
        out = fft_soft_border(arr, mode="low_pass", cutoff=0.3)
        assert np.isnan(out[3, 3])

    def test_high_pass_suppresses_constant_image(self):
        # Constant image after high-pass: DC is removed from windowed signal,
        # mean_val is added back, so output should still be near-constant
        arr = _uniform(32, 32, val=2.5)
        out = fft_soft_border(arr, mode="high_pass", cutoff=0.1)
        np.testing.assert_allclose(out, arr, atol=1e-8)

    def test_invalid_border_frac_raises(self):
        with pytest.raises(ValueError, match="border_frac"):
            fft_soft_border(np.ones((16, 16)), border_frac=0.6)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            fft_soft_border(np.ones((16, 16)), mode="band_pass")


# ── fft_magnitude ─────────────────────────────────────────────────────────────

class TestFFTMagnitude:
    def test_returns_three_arrays(self):
        arr = np.random.RandomState(0).randn(32, 32)
        mag, qx, qy = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert mag.ndim == 2
        assert qx.ndim == 1
        assert qy.ndim == 1

    def test_magnitude_nonnegative(self):
        arr = np.random.RandomState(1).randn(32, 32)
        mag, qx, qy = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert float(np.min(mag)) >= 0.0

    def test_dc_at_centre_of_qx(self):
        # After fftshift, the DC term is at index Nx//2 → qx[Nx//2] ≈ 0
        Nx = 64
        arr = np.random.RandomState(2).randn(64, Nx)
        _, qx, _ = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert abs(float(qx[Nx // 2])) < 1e-6

    def test_dc_at_centre_of_qy(self):
        Ny = 64
        arr = np.random.RandomState(3).randn(Ny, 64)
        _, _, qy = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert abs(float(qy[Ny // 2])) < 1e-6

    def test_axis_physical_units(self):
        # pixel_size_x_m = 0.25e-9 m → 0.25 nm → Nyquist at 1/(2*0.25) = 2 nm^-1
        Nx = 64
        px_m = 0.25e-9
        arr = np.random.RandomState(4).randn(64, Nx)
        _, qx, _ = fft_magnitude(arr, pixel_size_x_m=px_m, pixel_size_y_m=px_m)
        px_nm = px_m * 1e9
        nyquist = 1.0 / (2.0 * px_nm)
        # qx extremes should be near ±Nyquist
        assert abs(float(qx[-1])) == pytest.approx(nyquist, rel=0.1)

    def test_full_image_shape_matches(self):
        Ny, Nx = 24, 32
        arr = np.random.RandomState(5).randn(Ny, Nx)
        mag, qx, qy = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert mag.shape == (Ny, Nx)
        assert qx.shape == (Nx,)
        assert qy.shape == (Ny,)

    def test_log_scale_false_gives_linear_amplitude(self):
        arr = np.random.RandomState(6).randn(32, 32)
        mag_log, _, _ = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10, log_scale=True)
        mag_lin, _, _ = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10, log_scale=False)
        # log_scale output should be smaller (log1p compresses range)
        assert float(np.max(mag_log)) < float(np.max(mag_lin))

    def test_nan_input_runs_without_error(self):
        arr = np.ones((32, 32), dtype=np.float64)
        arr[5, 5] = np.nan
        mag, qx, qy = fft_magnitude(arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10)
        assert mag.shape == (32, 32)
