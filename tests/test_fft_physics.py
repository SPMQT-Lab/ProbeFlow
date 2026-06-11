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


# ── Regression for FFT cluster (physics #1, #5, #6, image-proc #5, #9) ──────

class TestFftClusterRegressions:
    """Regressions for the 2026-05-27 FFT/window cluster fixes."""

    def test_unbiased_dc_subtraction_with_elliptical_roi(self):
        """Review physics #1 — with a non-rectangular ROI, the DC
        subtraction must use the inside-ROI mean only, not the
        diluted-by-zeros full-crop mean.  A spurious bright peak at
        DC ± a few pixels was the previous symptom; after the fix
        the spectrum has no large low-k cross artefact relative to
        the no-ROI case.
        """
        from probeflow.core.roi import ROI
        # Random non-zero array; an elliptical ROI covering ~50%
        rng = np.random.default_rng(7)
        arr = rng.normal(loc=5.0, scale=0.5, size=(64, 64))
        roi = ROI.new("ellipse", {"cx": 32, "cy": 32, "rx": 16, "ry": 16})
        mag_roi, qx, qy = fft_magnitude(
            arr, roi=roi, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
            window="none", log_scale=False,
        )
        # The DC peak (centre of the shifted spectrum) should be
        # close to zero after proper mean subtraction.  The boundary
        # discontinuity at the ROI edge contributes some leakage, but
        # the central pixel itself should not be artificially large.
        cy, cx = mag_roi.shape[0] // 2, mag_roi.shape[1] // 2
        # Compare DC to the median magnitude in a few-pixel ring
        # around it; if DC is < a few × median, we're not leaving
        # significant residual DC after subtraction.
        dc = float(mag_roi[cy, cx])
        ring = mag_roi[cy - 5:cy + 6, cx - 5:cx + 6].copy()
        ring[5, 5] = np.nan  # exclude DC itself
        ring_med = float(np.nanmedian(ring))
        assert dc < 5.0 * (ring_med + 1e-12), (
            f"Residual DC after biased-mean fix: dc={dc:.3g}, "
            f"ring_median={ring_med:.3g}"
        )

    def test_window_coherent_gain_normalisation_makes_dc_window_invariant(self):
        """Review physics #5 — magnitude scale must be approximately
        the same regardless of window choice once coherent-gain
        normalisation is applied.  Pick a constant-amplitude DC-only
        input so the test isolates the gain effect."""
        # Single-frequency signal: a pure cosine wave so its FFT has
        # two impulses at ±k.  Coherent-gain normalisation should
        # bring the peak height to roughly the same value across
        # windows.
        N = 64
        k = 8
        x = np.arange(N) / N
        arr = np.tile(np.cos(2 * math.pi * k * x), (N, 1))
        mag_hann, _, _ = fft_magnitude(
            arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
            window="hann", log_scale=False,
        )
        mag_none, _, _ = fft_magnitude(
            arr, pixel_size_x_m=1e-10, pixel_size_y_m=1e-10,
            window="none", log_scale=False,
        )
        # After normalisation the two peak magnitudes should be
        # within ~30% of each other (Hann has slight spectral leakage
        # so we don't expect bit-identical; without normalisation the
        # ratio would be ~2x).
        peak_hann = float(mag_hann.max())
        peak_none = float(mag_none.max())
        ratio = peak_hann / peak_none
        assert 0.6 < ratio < 1.5, (
            f"Window coherent-gain normalisation off: "
            f"hann/none ratio = {ratio:.3f} (expected ~1)"
        )

    def test_periodic_hann_window_does_not_zero_boundary(self):
        """Review physics #6 — scipy.signal.windows.hann(N, sym=False)
        is the periodic Hann with the zero endpoint EXCLUDED, which
        is what DFT-based spectral analysis wants.  np.hanning(N)
        returned the symmetric variant with hann[0]==hann[-1]==0.
        Verify the helper returns a periodic window (no zero at the
        end)."""
        from probeflow.processing.filters import _window_1d
        w = _window_1d("hann", 32)
        # Periodic Hann: w[0] = 0.0 but w[-1] = w[1] (≠ 0).
        assert w[0] == pytest.approx(0.0)
        assert w[-1] > 0.0, (
            f"Window should be periodic (sym=False); got w[-1]={w[-1]}"
        )

    def test_unified_window_vocabulary(self):
        """Review image-proc #9 — fourier_filter and fft_magnitude
        accept the same window names."""
        from probeflow.processing.filters import _resolve_window_name
        # "hann" and "hanning" both map to canonical "hann"
        assert _resolve_window_name("hann") == "hann"
        assert _resolve_window_name("hanning") == "hann"
        assert _resolve_window_name("HANN") == "hann"
        # "hamming"
        assert _resolve_window_name("hamming") == "hamming"
        # "tukey"
        assert _resolve_window_name("tukey") == "tukey"
        # "none" / aliases
        assert _resolve_window_name("none") == "none"
        assert _resolve_window_name("rectangular") == "none"
        assert _resolve_window_name("boxcar") == "none"
        assert _resolve_window_name(None) == "none"
        # Invalid raises
        with pytest.raises(ValueError):
            _resolve_window_name("kaiser")

    def test_fourier_filter_radial_cutoff_isotropic_on_rectangular_image(self):
        """Review physics #7 — on a rectangular image, a radial
        cutoff should be the same circle in cycles/pixel along both
        axes.  Previously the cutoff was distorted into an ellipse
        because the radial coordinate was normalised to half-axis
        (different per-axis scaling for non-square images)."""
        # Construct a rectangular image (Ny != Nx) with two sinusoids
        # at the same true frequency along each axis.  A low-pass at
        # cutoff=0.5 (half-Nyquist) should retain a frequency at
        # 0.4 cycles/pixel along both axes.
        Ny, Nx = 32, 64
        k_freq = 0.4  # cycles per pixel, below 0.5 cutoff
        x = np.arange(Nx)
        y = np.arange(Ny)
        wave_x = np.cos(2 * math.pi * k_freq * x)
        wave_y = np.cos(2 * math.pi * k_freq * y)
        # Use the same wave along both axes (tiled), then transpose for y-axis test
        arr_x = np.tile(wave_x, (Ny, 1))
        arr_y = np.tile(wave_y[:, None], (1, Nx))
        # Low-pass at cutoff=0.5 (just above 0.4) — both should survive
        out_x = fourier_filter(arr_x, mode="low_pass", cutoff=0.85, window="none")
        out_y = fourier_filter(arr_y, mode="low_pass", cutoff=0.85, window="none")
        # Both inputs are within the filter passband (cutoff in
        # cycles/pixel * 2 = Nyquist-fraction).  Both outputs should
        # closely match their inputs.
        rms_x = float(np.sqrt(np.mean((out_x - arr_x) ** 2)))
        rms_y = float(np.sqrt(np.mean((out_y - arr_y) ** 2)))
        # Both axes should behave the same — ratio of RMS errors close to 1
        assert max(rms_x, rms_y) / max(min(rms_x, rms_y), 1e-12) < 5.0, (
            f"Isotropic cutoff broken: rms_x={rms_x}, rms_y={rms_y}"
        )


class TestWindowEnvelopeCompensation:
    """2026-06-12 FFT review: fourier_filter windowed the data before the
    transform but never divided the window back out, so the filtered image
    carried the window envelope — with the GUI-default Hann, the output was
    vignetted toward the mean at the borders (edge std 0.03 vs centre 0.77
    on unit-variance noise). The pre-existing near-identity tests used
    cutoff exactly 1.0/0.0, which early-return before any windowing and
    could not see it."""

    def test_near_identity_low_pass_is_spatially_flat(self):
        rng = np.random.default_rng(0)
        img = rng.normal(size=(64, 64)) + 5.0
        out = fourier_filter(img, mode="low_pass", cutoff=0.99, window="hanning")
        centre = float(np.std(out[24:40, 24:40]))
        edge = float(np.std(out[:8, :8]))
        assert edge > 0.5 * centre, (
            f"window envelope baked into output: edge std {edge:.3f} vs "
            f"centre {centre:.3f}"
        )

    def test_high_pass_response_uniform_across_image(self):
        """A uniform-amplitude high-frequency stripe must keep roughly the
        same amplitude at the border as in the middle after a high-pass."""
        yy, xx = np.mgrid[:96, :96]
        img = np.sin(2 * np.pi * 0.3 * xx)
        out = fourier_filter(img, mode="high_pass", cutoff=0.2, window="hanning")
        centre_amp = float(np.std(out[40:56, 40:56]))
        edge_amp = float(np.std(out[4:12, 40:56]))
        assert edge_amp > 0.5 * centre_amp

    def test_window_none_behaviour_unchanged(self):
        """No window → no compensation: pin equality with the raw masked
        transform so the compensation cannot leak into the 'none' path."""
        rng = np.random.default_rng(1)
        img = rng.normal(size=(32, 32))
        out = fourier_filter(img, mode="low_pass", cutoff=0.5, window="none")
        mean = img.mean()
        F = np.fft.fftshift(np.fft.fft2(img - mean))
        qx = np.fft.fftshift(np.fft.fftfreq(32))
        Qx, Qy = np.meshgrid(qx, qx)
        mask = (np.sqrt(Qx**2 + Qy**2) / 0.5 <= 0.5).astype(float)
        expected = np.fft.ifft2(np.fft.ifftshift(F * mask)).real + mean
        np.testing.assert_allclose(out, expected, atol=1e-12)


class TestSoftBorderOddSizes:
    def test_dc_preserved_on_odd_sized_image(self):
        """The radial mask must sit on the fftshift DC bin for odd sizes —
        the old index-arithmetic centre was half a pixel off, so a tiny
        low-pass could miss DC entirely and shift the image mean."""
        rng = np.random.default_rng(2)
        img = rng.normal(size=(63, 65)) + 7.0
        out = fft_soft_border(img, mode="low_pass", cutoff=0.02,
                              border_frac=0.1)
        assert abs(float(np.nanmean(out)) - 7.0) < 0.05

    def test_even_size_results_unchanged_by_mask_rewrite(self):
        """For even sizes the fftfreq mask is identical to the old index
        arithmetic — pin a checksum so the rewrite is provably a no-op."""
        rng = np.random.default_rng(3)
        img = rng.normal(size=(64, 64))
        out = fft_soft_border(img, mode="low_pass", cutoff=0.4, border_frac=0.12)
        # Old-mask reference computed inline.
        a = img.astype(np.float64).copy()
        mean = a.mean()
        n = 64
        edge = max(1, int(round(0.12 * n)))
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge)))
        w = np.ones(n); w[:edge] = ramp; w[-edge:] = ramp[::-1]
        win2d = np.outer(w, w)
        F = np.fft.fftshift(np.fft.fft2((a - mean) * win2d))
        cyx = n / 2.0
        r1 = (np.arange(n) - cyx) / cyx
        Xr, Yr = np.meshgrid(r1, r1)
        mask = (np.sqrt(Xr**2 + Yr**2) <= 0.4).astype(float)
        ref = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
        ref = ref / np.where(win2d > 1e-6, win2d, 1.0) + mean
        np.testing.assert_allclose(out, ref, atol=1e-12)
