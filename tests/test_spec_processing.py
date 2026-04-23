"""Tests for probeflow.spec_processing — spectroscopy processing functions."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.spec_processing import (
    average_spectra,
    crop,
    current_histogram,
    normalize,
    numeric_derivative,
    smooth_spectrum,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_signal():
    x = np.linspace(0, 2 * np.pi, 200)
    return x, np.sin(x)


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 500)
    y = np.sin(4 * np.pi * x) + rng.normal(0, 0.3, 500)
    return x, y


# ─── smooth_spectrum ─────────────────────────────────────────────────────────

class TestSmoothSpectrum:
    def test_savgol_reduces_noise(self, noisy_signal):
        _, y = noisy_signal
        smoothed = smooth_spectrum(y, method="savgol", window_length=21, polyorder=3)
        assert smoothed.std() < y.std()

    def test_savgol_same_length(self, noisy_signal):
        _, y = noisy_signal
        smoothed = smooth_spectrum(y)
        assert len(smoothed) == len(y)

    def test_gaussian_same_length(self, noisy_signal):
        _, y = noisy_signal
        smoothed = smooth_spectrum(y, method="gaussian", sigma=3.0)
        assert len(smoothed) == len(y)

    def test_boxcar_same_length(self, noisy_signal):
        _, y = noisy_signal
        smoothed = smooth_spectrum(y, method="boxcar", n=7)
        assert len(smoothed) == len(y)

    def test_flat_signal_unchanged(self):
        y = np.ones(100)
        smoothed = smooth_spectrum(y)
        assert np.allclose(smoothed, 1.0, atol=1e-10)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            smooth_spectrum(np.ones(50), method="fft")


# ─── numeric_derivative ──────────────────────────────────────────────────────

class TestNumericDerivative:
    def test_constant_zero_derivative(self):
        x = np.linspace(0, 1, 100)
        y = np.full(100, 3.14)
        dy = numeric_derivative(x, y)
        assert np.allclose(dy, 0.0, atol=1e-10)

    def test_linear_unit_derivative(self):
        x = np.linspace(0, 5, 500)
        y = 2.0 * x + 1.0
        dy = numeric_derivative(x, y)
        # Central differences are exact for linear functions
        assert np.allclose(dy, 2.0, atol=1e-8)

    def test_sine_cosine(self, sine_signal):
        x, y = sine_signal
        dy = numeric_derivative(x, y)
        expected = np.cos(x)
        # Interior points should match cos(x) closely
        assert np.allclose(dy[5:-5], expected[5:-5], atol=0.02)

    def test_same_length(self, sine_signal):
        x, y = sine_signal
        dy = numeric_derivative(x, y)
        assert len(dy) == len(x)


# ─── normalize ───────────────────────────────────────────────────────────────

class TestNormalize:
    def test_max_peak_is_one(self):
        y = np.array([0.0, 3.0, -1.0, 2.0])
        n = normalize(y, method="max")
        assert n.max() == pytest.approx(1.0)

    def test_minmax_range(self):
        y = np.array([1.0, 3.0, 5.0, 7.0])
        n = normalize(y, method="minmax")
        assert n.min() == pytest.approx(0.0)
        assert n.max() == pytest.approx(1.0)

    def test_zscore_mean_std(self):
        rng = np.random.default_rng(1)
        y = rng.normal(5.0, 2.0, 1000)
        n = normalize(y, method="zscore")
        assert abs(n.mean()) < 0.01
        assert abs(n.std() - 1.0) < 0.01

    def test_constant_array_max(self):
        y = np.full(50, 7.0)
        n = normalize(y, method="max")
        assert np.all(n == pytest.approx(1.0))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            normalize(np.ones(10), method="rms")


# ─── crop ────────────────────────────────────────────────────────────────────

class TestCrop:
    def test_basic_crop(self):
        x = np.linspace(-1, 1, 100)
        y = x ** 2
        xc, yc = crop(x, y, -0.5, 0.5)
        assert xc.min() >= -0.5
        assert xc.max() <= 0.5
        assert len(xc) == len(yc)

    def test_no_crop_returns_full(self):
        x = np.linspace(0, 1, 50)
        y = np.ones(50)
        xc, yc = crop(x, y, 0.0, 1.0)
        assert len(xc) == 50

    def test_empty_crop(self):
        x = np.linspace(0, 1, 50)
        y = np.ones(50)
        xc, yc = crop(x, y, 2.0, 3.0)
        assert len(xc) == 0
        assert len(yc) == 0


# ─── average_spectra ─────────────────────────────────────────────────────────

class TestAverageSpectra:
    def test_single_spectrum(self):
        y = np.array([1.0, 2.0, 3.0])
        avg = average_spectra([y])
        assert np.allclose(avg, y)

    def test_two_spectra(self):
        a = np.array([0.0, 2.0, 4.0])
        b = np.array([2.0, 4.0, 6.0])
        avg = average_spectra([a, b])
        assert np.allclose(avg, [1.0, 3.0, 5.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            average_spectra([])

    def test_output_shape(self):
        spectra = [np.random.rand(200) for _ in range(5)]
        avg = average_spectra(spectra)
        assert avg.shape == (200,)


# ─── current_histogram ───────────────────────────────────────────────────────

class TestCurrentHistogram:
    def test_return_shapes(self):
        data = np.linspace(-1e-9, 1e-9, 500)
        edges, counts = current_histogram(data, bins=50)
        assert len(edges) == 51
        assert len(counts) == 50

    def test_total_counts(self):
        data = np.ones(300) * 1e-10
        edges, counts = current_histogram(data, bins=20)
        assert counts.sum() == 300

    def test_nan_ignored(self):
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
        edges, counts = current_histogram(data, bins=10)
        assert counts.sum() == 3

    def test_bimodal_two_peaks(self):
        # Two groups of values well separated
        data = np.concatenate([np.full(100, 1e-10), np.full(100, 5e-10)])
        edges, counts = current_histogram(data, bins=100)
        # The two groups should be in different halves of the histogram
        mid = len(counts) // 2
        assert counts[:mid].sum() > 0
        assert counts[mid:].sum() > 0
