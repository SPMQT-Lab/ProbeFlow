"""Contract tests for spectroscopy processing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.spectroscopy import (
    average_spectra,
    crop,
    current_histogram,
    normalize,
    numeric_derivative,
    smooth_spectrum,
)


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


def test_smooth_spectrum_methods_preserve_domain_invariants(noisy_signal):
    _, y = noisy_signal

    savgol = smooth_spectrum(y, method="savgol", window_length=21, polyorder=3)
    gaussian = smooth_spectrum(y, method="gaussian", sigma=3.0)
    boxcar = smooth_spectrum(y, method="boxcar", n=7)

    for smoothed in (savgol, gaussian, boxcar):
        assert len(smoothed) == len(y)
    assert savgol.std() < y.std()
    assert np.allclose(smooth_spectrum(np.ones(100)), 1.0, atol=1e-10)
    assert np.allclose(
        smooth_spectrum(np.ones(100), method="boxcar", n=7),
        1.0,
        atol=1e-12,
    )
    short = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(smooth_spectrum(short, method="savgol", polyorder=3), short)


def test_smooth_spectrum_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        smooth_spectrum(np.ones(50), method="fft")


def test_numeric_derivative_matches_known_functions_and_reverse_sweeps(sine_signal):
    x = np.linspace(0, 1, 100)
    constant = numeric_derivative(x, np.full(100, 3.14))
    assert np.allclose(constant, 0.0, atol=1e-10)

    x_linear = np.linspace(0, 5, 500)
    linear = numeric_derivative(x_linear, 2.0 * x_linear + 1.0)
    assert np.allclose(linear, 2.0, atol=1e-8)

    x_sine, y_sine = sine_signal
    sine = numeric_derivative(x_sine, y_sine)
    assert len(sine) == len(x_sine)
    assert np.allclose(sine[5:-5], np.cos(x_sine)[5:-5], atol=0.02)

    x_desc = np.linspace(5.0, 0.0, 100)
    descending = numeric_derivative(x_desc, 3.0 * x_desc + 1.0)
    assert np.allclose(descending, 3.0, atol=1e-8)


def test_numeric_derivative_rejects_duplicate_or_non_monotonic_x_values():
    bad_cases = [
        (np.array([0.0, 1.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0, 1.0, 0.0]), np.array([0.0, 1.0, 4.0, 1.0, 0.0])),
    ]
    for x, y in bad_cases:
        with pytest.raises(ValueError, match="monotonic"):
            numeric_derivative(x, y)


def test_normalize_methods_enforce_expected_scale_contracts():
    assert normalize(np.array([0.0, 3.0, -1.0, 2.0]), method="max").max() == pytest.approx(1.0)

    minmax = normalize(np.array([1.0, 3.0, 5.0, 7.0]), method="minmax")
    assert minmax.min() == pytest.approx(0.0)
    assert minmax.max() == pytest.approx(1.0)

    rng = np.random.default_rng(1)
    zscore = normalize(rng.normal(5.0, 2.0, 1000), method="zscore")
    assert abs(zscore.mean()) < 0.01
    assert abs(zscore.std() - 1.0) < 0.01

    constant = normalize(np.full(50, 7.0), method="max")
    assert np.all(constant == pytest.approx(1.0))
    np.testing.assert_allclose(
        normalize(np.array([0.0, -2.0, -4.0]), method="setpoint"),
        [-0.0, 1.0, 2.0],
    )


def test_normalize_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        normalize(np.ones(10), method="rms")


def test_crop_handles_bounds_swaps_full_range_and_empty_results():
    x = np.linspace(-1, 1, 100)
    y = x**2

    cropped_x, cropped_y = crop(x, y, -0.5, 0.5)
    assert cropped_x.min() >= -0.5
    assert cropped_x.max() <= 0.5
    assert len(cropped_x) == len(cropped_y)

    full_x, full_y = crop(x, y, -1.0, 1.0)
    np.testing.assert_array_equal(full_x, x)
    np.testing.assert_array_equal(full_y, y)

    empty_x, empty_y = crop(x, y, 2.0, 3.0)
    assert len(empty_x) == 0
    assert len(empty_y) == 0

    forward_x, forward_y = crop(x, y, 0.2, 0.8)
    reverse_x, reverse_y = crop(x, y, 0.8, 0.2)
    np.testing.assert_array_equal(forward_x, reverse_x)
    np.testing.assert_array_equal(forward_y, reverse_y)


def test_average_spectra_contract_for_single_multi_and_shape():
    single = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(average_spectra([single]), single)

    avg = average_spectra([np.array([0.0, 2.0, 4.0]), np.array([2.0, 4.0, 6.0])])
    np.testing.assert_allclose(avg, [1.0, 3.0, 5.0])

    spectra = [np.random.rand(200) for _ in range(5)]
    assert average_spectra(spectra).shape == (200,)


def test_average_spectra_rejects_empty_or_mismatched_inputs():
    with pytest.raises(ValueError):
        average_spectra([])
    with pytest.raises(ValueError, match="same length"):
        average_spectra([np.ones(100), np.ones(150)])


def test_current_histogram_counts_finite_values_and_separates_bimodal_groups():
    counts, edges = current_histogram(np.linspace(-1e-9, 1e-9, 500), bins=50)
    assert len(counts) == 50
    assert len(edges) == 51

    nan_counts, _ = current_histogram(np.array([1.0, 2.0, np.nan, 3.0, np.nan]), bins=10)
    assert nan_counts.sum() == 3

    data = np.concatenate([np.full(100, 1e-10), np.full(100, 5e-10)])
    bimodal_counts, _ = current_histogram(data, bins=100)
    mid = len(bimodal_counts) // 2
    assert bimodal_counts[:mid].sum() > 0
    assert bimodal_counts[mid:].sum() > 0
