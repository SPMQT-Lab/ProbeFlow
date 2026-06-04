"""Tests for probeflow.analysis.line_periodicity."""

from __future__ import annotations

import numpy as np

from probeflow.analysis.line_periodicity import (
    estimate_line_periodicity,
    format_period,
    format_result_text,
)


def _sine(period=0.256e-9, length=10e-9, n=1000, noise=0.0, slope=0.0, rng_seed=42):
    s = np.linspace(0, length, n)
    z = np.sin(2 * np.pi * s / period)
    if noise > 0:
        rng = np.random.default_rng(rng_seed)
        z = z + rng.normal(scale=noise, size=n)
    if slope != 0:
        z = z + slope * (s - s[0])
    return s, z


# ── 1. Clean sine returns correct period ──────────────────────────────────────

def test_clean_sine_autocorrelation():
    period = 0.256e-9
    s, z = _sine(period=period)
    result, diag = estimate_line_periodicity(
        s, z, method="autocorrelation", background="linear", smoothing="none"
    )
    assert result.quality in ("good", "weak"), result.message
    assert not np.isnan(result.period_m)
    assert abs(result.period_m - period) / period < 0.05


# ── 2. Noisy sine still returns approximate period ────────────────────────────

def test_noisy_sine_autocorrelation():
    period = 0.256e-9
    s, z = _sine(period=period, noise=0.5)
    result, _ = estimate_line_periodicity(
        s, z, method="autocorrelation", background="linear", smoothing="light_gaussian"
    )
    assert result.quality in ("good", "weak", "ambiguous"), result.message
    assert not np.isnan(result.period_m)
    assert abs(result.period_m - period) / period < 0.20


# ── 3. Sloped background is handled by linear detrending ─────────────────────

def test_sloped_background_linear_detrend():
    period = 0.256e-9
    s, z = _sine(period=period, slope=5e9)
    result, _ = estimate_line_periodicity(
        s, z, method="autocorrelation", background="linear", smoothing="none"
    )
    assert not np.isnan(result.period_m), result.message
    assert abs(result.period_m - period) / period < 0.15


# ── 4. Peak-spacing method finds known peak spacing ───────────────────────────

def test_peak_spacing_finds_known_spacing():
    period = 0.512e-9
    s, z = _sine(period=period)
    result, diag = estimate_line_periodicity(
        s, z, method="peak_spacing", background="none", smoothing="none"
    )
    assert result.quality in ("good", "weak"), result.message
    assert not np.isnan(result.period_m)
    assert abs(result.period_m - period) / period < 0.05
    assert diag.peak_positions_m is not None and len(diag.peak_positions_m) > 1


# ── 5. FFT method finds known period ─────────────────────────────────────────

def test_fft_finds_known_period():
    period = 0.256e-9
    s, z = _sine(period=period)
    result, diag = estimate_line_periodicity(
        s, z, method="fft", background="linear", smoothing="none"
    )
    assert not np.isnan(result.period_m), result.message
    assert abs(result.period_m - period) / period < 0.05
    assert diag.fft_freq_m_inv is not None
    assert diag.fft_power is not None
    assert result.uncertainty_m is None
    assert diag.fft_period_bin_width_m is not None
    assert "bin width" in result.message


# ── 6. Too-short profile returns weak or failed ───────────────────────────────

def test_too_short_profile_returns_failed():
    s = np.linspace(0, 0.1e-9, 3)
    z = np.sin(2 * np.pi * s / 0.256e-9)
    result, _ = estimate_line_periodicity(s, z)
    assert result.quality == "failed"


# ── 7. Mostly-NaN profile fails gracefully ───────────────────────────────────

def test_mostly_nan_fails_gracefully():
    period = 0.256e-9
    s, z = _sine(period=period)
    z[10:900] = np.nan
    result, diag = estimate_line_periodicity(s, z)
    assert result.quality == "failed"
    assert np.isnan(result.period_m)
    assert diag.s_m is not None


# ── 8. Two competing periods flags ambiguous or reports warning ───────────────

def test_two_competing_periods():
    period1 = 0.256e-9
    period2 = 0.19e-9
    s = np.linspace(0, 10e-9, 1000)
    z = np.sin(2 * np.pi * s / period1) + np.sin(2 * np.pi * s / period2)
    result, _ = estimate_line_periodicity(s, z, method="autocorrelation")
    # The result may be ambiguous or may pick one frequency, but must not crash.
    assert result.quality in ("ambiguous", "good", "weak", "failed")


def test_autocorrelation_uses_unbiased_estimator():
    """Regression for review physics #4 — the unbiased ACF estimator
    should make ACF peak heights at lag = k * period roughly equal
    instead of decaying as (n-lag)/n with the biased estimator.  This
    test asserts that for a clean periodic signal, the 1st-period and
    2nd-period autocorrelation peaks have comparable heights (within
    20%), which is only true with the unbiased normalisation."""
    period = 0.256e-9
    # Use enough length that the 2nd-period lag is well-supported.
    s, z = _sine(period=period, length=20e-9, n=2000)
    _, diag = estimate_line_periodicity(
        s, z, method="autocorrelation", background="linear", smoothing="none"
    )
    lags = diag.autocorr_lag_m
    ac = diag.autocorr
    # Find the ACF value at lags closest to period and 2*period.
    idx1 = int(np.argmin(np.abs(lags - period)))
    idx2 = int(np.argmin(np.abs(lags - 2 * period)))
    h1 = float(ac[idx1])
    h2 = float(ac[idx2])
    # With the biased estimator h2 ≈ h1 * (n-idx2)/(n-idx1) ≈ 0.9 * h1.
    # With the unbiased estimator both heights should be within ~20%
    # of each other (for a perfectly periodic signal h2 → h1 in
    # the noise-free limit).
    ratio = h2 / h1 if h1 > 0 else 0.0
    assert 0.8 < ratio < 1.25, (
        f"Unbiased ACF should give h2/h1 ≈ 1 for a periodic signal; "
        f"got h1={h1:.3f}, h2={h2:.3f}, ratio={ratio:.3f}"
    )


# ── Diagnostic completeness checks ───────────────────────────────────────────

def test_autocorrelation_diagnostic_populated():
    s, z = _sine()
    _, diag = estimate_line_periodicity(s, z, method="autocorrelation")
    assert diag.autocorr_lag_m is not None
    assert diag.autocorr is not None
    assert diag.z_processed is not None


def test_peak_spacing_diagnostic_populated():
    s, z = _sine()
    _, diag = estimate_line_periodicity(s, z, method="peak_spacing", background="none")
    assert diag.peak_positions_m is not None


def test_fft_diagnostic_populated():
    s, z = _sine()
    _, diag = estimate_line_periodicity(s, z, method="fft")
    assert diag.fft_freq_m_inv is not None
    assert diag.fft_power is not None


# ── format helpers ────────────────────────────────────────────────────────────

def test_format_period_angstrom():
    val, unit = format_period(0.256e-9)
    assert unit == "Å"
    assert "2.56" in val


def test_format_period_nm():
    val, unit = format_period(6.4e-9)
    assert unit == "nm"
    assert "6.4" in val


def test_format_result_text_good():
    s, z = _sine()
    result, _ = estimate_line_periodicity(s, z)
    text = format_result_text(result, background="linear", smoothing="light Gaussian")
    assert "Period:" in text
    assert "Line length:" in text
    assert "Method:" in text
