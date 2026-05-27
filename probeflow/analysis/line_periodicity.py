"""Line-profile periodicity estimation.

Estimates a repeat distance from a 1-D profile extracted along a line ROI.
Three methods are provided: autocorrelation (default), peak spacing, and FFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class PeriodicityResult:
    method: str
    period_m: float
    uncertainty_m: float | None
    line_length_m: float
    n_periods: float
    n_samples: int
    quality: str
    message: str


@dataclass(frozen=True)
class PeriodicityDiagnostic:
    s_m: np.ndarray
    z_raw: np.ndarray
    z_processed: np.ndarray
    autocorr_lag_m: np.ndarray | None = None
    autocorr: np.ndarray | None = None
    peak_positions_m: np.ndarray | None = None
    fft_freq_m_inv: np.ndarray | None = None
    fft_power: np.ndarray | None = None


def estimate_line_periodicity(
    s_m: np.ndarray,
    z: np.ndarray,
    *,
    method: str = "autocorrelation",
    background: str = "linear",
    smoothing: str = "light_gaussian",
    min_period_m: float | None = None,
    max_period_m: float | None = None,
) -> tuple[PeriodicityResult, PeriodicityDiagnostic]:
    """Estimate the repeat distance in a 1-D line profile.

    Parameters
    ----------
    s_m:
        Distance coordinates along the line, in metres.
    z:
        Profile values at each point.
    method:
        ``"autocorrelation"`` (default), ``"peak_spacing"``, or ``"fft"``.
    background:
        Background removal: ``"none"``, ``"linear"``,
        ``"polynomial_2"``, or ``"moving_average"``.
    smoothing:
        Smoothing after background removal: ``"none"``,
        ``"light_gaussian"``, or ``"savitzky_golay"``.
    min_period_m, max_period_m:
        Optional period search bounds, in metres.
    """
    s_m = np.asarray(s_m, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if s_m.shape != z.shape or s_m.ndim != 1:
        raise ValueError("s_m and z must be 1-D arrays of the same length")

    n = len(s_m)
    line_length_m = float(s_m[-1] - s_m[0]) if n > 1 else 0.0

    def _failed(msg: str, z_proc=None) -> tuple[PeriodicityResult, PeriodicityDiagnostic]:
        z_p = z_proc if z_proc is not None else z.copy()
        return (
            PeriodicityResult(
                method=method,
                period_m=float("nan"),
                uncertainty_m=None,
                line_length_m=line_length_m,
                n_periods=0.0,
                n_samples=n,
                quality="failed",
                message=msg,
            ),
            PeriodicityDiagnostic(s_m=s_m, z_raw=z.copy(), z_processed=z_p),
        )

    if n < 4:
        return _failed("Profile too short (fewer than 4 samples).")

    finite_mask = np.isfinite(z)
    if finite_mask.sum() < max(4, n // 2):
        return _failed(
            f"Too many non-finite values ({(~finite_mask).sum()}/{n}).",
            np.where(finite_mask, z, np.nan),
        )

    z_clean = _interpolate_nans(s_m, z)
    z_proc = _remove_background(s_m, z_clean, background)
    z_proc = _smooth(z_proc, smoothing)

    sample_spacing = line_length_m / max(n - 1, 1)

    try:
        if method == "autocorrelation":
            result, extra = _autocorrelation_method(
                s_m, z_proc, sample_spacing, line_length_m, n,
                min_period_m=min_period_m, max_period_m=max_period_m,
            )
        elif method == "peak_spacing":
            result, extra = _peak_spacing_method(
                s_m, z_proc, sample_spacing, line_length_m, n,
                min_period_m=min_period_m, max_period_m=max_period_m,
            )
        elif method == "fft":
            result, extra = _fft_method(
                s_m, z_proc, sample_spacing, line_length_m, n,
                min_period_m=min_period_m, max_period_m=max_period_m,
            )
        else:
            return _failed(f"Unknown method {method!r}.", z_proc)
    except Exception as exc:
        return _failed(str(exc), z_proc)

    diag = PeriodicityDiagnostic(s_m=s_m, z_raw=z.copy(), z_processed=z_proc, **extra)
    return result, diag


# ── preprocessing ─────────────────────────────────────────────────────────────

def _interpolate_nans(s_m: np.ndarray, z: np.ndarray) -> np.ndarray:
    finite = np.isfinite(z)
    if finite.all():
        return z.copy()
    z_out = z.copy()
    nan_idx = np.where(~finite)[0]
    fin_idx = np.where(finite)[0]
    z_out[nan_idx] = np.interp(s_m[nan_idx], s_m[fin_idx], z[fin_idx])
    return z_out


def _remove_background(s_m: np.ndarray, z: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return z.copy()
    if method == "linear":
        return signal.detrend(z, type="linear")
    if method == "polynomial_2":
        x = (s_m - s_m[0]) / (s_m[-1] - s_m[0]) if s_m[-1] != s_m[0] else s_m - s_m[0]
        coeffs = np.polyfit(x, z, 2)
        return z - np.polyval(coeffs, x)
    if method == "moving_average":
        n = len(z)
        win = max(3, n // 5)
        win = win + 1 if win % 2 == 0 else win
        bg = np.convolve(z, np.ones(win) / win, mode="same")
        half = win // 2
        bg[:half] = bg[half]
        bg[n - half - 1:] = bg[n - half - 2]
        return z - bg
    raise ValueError(f"Unknown background method {method!r}")


def _smooth(z: np.ndarray, method: str) -> np.ndarray:
    n = len(z)
    if method == "none":
        return z.copy()
    if method == "light_gaussian":
        return ndimage.gaussian_filter1d(z, sigma=1.5)
    if method == "savitzky_golay":
        win = min(11, max(5, n // 5))
        win = win + 1 if win % 2 == 0 else win
        if win > n:
            return z.copy()
        return signal.savgol_filter(z, window_length=win, polyorder=2)
    raise ValueError(f"Unknown smoothing method {method!r}")


# ── methods ───────────────────────────────────────────────────────────────────

def _autocorrelation_method(
    s_m, z_proc, sample_spacing, line_length_m, n, *, min_period_m, max_period_m
) -> tuple[PeriodicityResult, dict]:
    z_norm = z_proc - np.mean(z_proc)
    std = np.std(z_norm)
    if std > 0:
        z_norm /= std

    ac_full = np.correlate(z_norm, z_norm, mode="full")

    # Positive lags: ac_pos[i] corresponds to lag (i+1) * sample_spacing.
    # Review physics #4 (fixed 2026-05-28): the previous implementation
    # used the biased ACF (np.correlate / zero_lag_val), where amplitude
    # decays linearly as ``(n - k) / n`` with lag — even for a perfectly
    # periodic signal — so the peak finder biased toward the first
    # (shortest-lag) peak and the ambiguity check at higher lags
    # compared against artificially suppressed harmonics.  Use the
    # unbiased estimator: divide each lag by its overlap count.
    overlap_counts = n - np.arange(1, n, dtype=np.float64)
    ac_pos = ac_full[n:] / overlap_counts

    # Clip the trustworthy lag range so the noise-amplified far tail
    # (where overlap is tiny) cannot dominate.  Lags where the overlap
    # count is below n/8 are dropped from the search.
    min_overlap = max(int(n / 8), 2)
    ac_pos = np.where(overlap_counts >= min_overlap, ac_pos, -np.inf)

    lag_m = np.arange(1, n) * sample_spacing

    # Period bounds → restrict search range
    valid = np.ones(len(lag_m), dtype=bool)
    if min_period_m is not None:
        valid &= lag_m >= min_period_m
    if max_period_m is not None:
        valid &= lag_m <= max_period_m

    extra = {"autocorr_lag_m": lag_m, "autocorr": ac_pos}

    if valid.sum() < 3:
        return _make_failed("autocorrelation", line_length_m, n,
                            "No valid lag range for autocorrelation peak search."), extra

    ac_search = np.where(valid, ac_pos, -np.inf)
    peaks, props = signal.find_peaks(ac_search, prominence=0.05, height=0.1)

    if len(peaks) == 0:
        return _make_failed("autocorrelation", line_length_m, n,
                            "No autocorrelation peak found."), extra

    first_idx = peaks[0]
    prominence = float(props["prominences"][0])

    # Parabolic sub-sample refinement
    delta = 0.0
    if 0 < first_idx < len(ac_pos) - 1:
        y0, y1, y2 = ac_pos[first_idx - 1], ac_pos[first_idx], ac_pos[first_idx + 1]
        denom = 2 * y1 - y0 - y2
        if denom > 0:
            delta = 0.5 * (y0 - y2) / denom

    period_m = lag_m[first_idx] + delta * sample_spacing

    # FWHM uncertainty
    half_max = ac_pos[first_idx] / 2
    left, right = first_idx, first_idx
    while left > 0 and ac_pos[left] > half_max:
        left -= 1
    while right < len(ac_pos) - 1 and ac_pos[right] > half_max:
        right += 1
    fwhm = right - left
    uncertainty_m = (fwhm / 2) * sample_spacing if fwhm > 0 else None

    # Ambiguity check: second peak is not at a harmonic ratio
    quality, message = _quality_from_n_periods(line_length_m / period_m, prominence)
    if len(peaks) > 1:
        ratio = lag_m[peaks[1]] / period_m
        if 1.1 < ratio < 1.85:
            quality = "ambiguous"
            message = (
                f"Multiple competing peaks: "
                f"{_fmt_m(period_m)} and {_fmt_m(lag_m[peaks[1]])}."
            )

    return PeriodicityResult(
        method="autocorrelation",
        period_m=period_m,
        uncertainty_m=uncertainty_m,
        line_length_m=line_length_m,
        n_periods=line_length_m / period_m if period_m > 0 else 0.0,
        n_samples=n,
        quality=quality,
        message=message,
    ), extra


def _peak_spacing_method(
    s_m, z_proc, sample_spacing, line_length_m, n, *, min_period_m, max_period_m
) -> tuple[PeriodicityResult, dict]:
    min_dist = max(2, int(min_period_m / sample_spacing)) if min_period_m else 2
    prominence = np.std(z_proc) * 0.1

    peaks, _ = signal.find_peaks(z_proc, distance=min_dist, prominence=prominence)
    peak_positions = s_m[peaks] if len(peaks) > 0 else np.array([])
    extra = {"peak_positions_m": peak_positions}

    if len(peaks) < 2:
        return _make_failed("peak_spacing", line_length_m, n,
                            f"Only {len(peaks)} peak(s) found; need at least 2."), extra

    spacings = np.diff(peak_positions)
    if min_period_m is not None:
        spacings = spacings[spacings >= min_period_m]
    if max_period_m is not None:
        spacings = spacings[spacings <= max_period_m]

    if len(spacings) == 0:
        return _make_failed("peak_spacing", line_length_m, n,
                            "No spacings within the specified period bounds."), extra

    period_m = float(np.median(spacings))
    uncertainty_m = float(np.std(spacings)) if len(spacings) > 1 else None
    n_periods = line_length_m / period_m if period_m > 0 else 0.0
    cv = (float(np.std(spacings)) / period_m) if period_m > 0 and len(spacings) > 1 else 0.0

    if n_periods < 2:
        quality = "weak"
        message = f"{len(peaks)} peaks; only {n_periods:.1f} periods sampled."
    elif cv > 0.3:
        quality = "ambiguous"
        message = f"Large spread in peak spacings (CV={cv:.0%})."
    elif n_periods >= 3 and cv <= 0.1:
        quality = "good"
        message = f"{len(peaks)} peaks; median spacing {_fmt_m(period_m)}."
    else:
        quality = "weak"
        message = f"{len(peaks)} peaks; moderate spread in spacings (CV={cv:.0%})."

    return PeriodicityResult(
        method="peak_spacing",
        period_m=period_m,
        uncertainty_m=uncertainty_m,
        line_length_m=line_length_m,
        n_periods=n_periods,
        n_samples=n,
        quality=quality,
        message=message,
    ), extra


def _fft_method(
    s_m, z_proc, sample_spacing, line_length_m, n, *, min_period_m, max_period_m
) -> tuple[PeriodicityResult, dict]:
    z_dt = signal.detrend(z_proc, type="linear")
    z_windowed = z_dt * np.hanning(n)

    fft_vals = np.fft.rfft(z_windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=sample_spacing)

    # Exclude DC; apply period bounds as frequency bounds
    valid = freqs > 0
    if min_period_m is not None and min_period_m > 0:
        valid &= freqs <= 1.0 / min_period_m
    if max_period_m is not None and max_period_m > 0:
        valid &= freqs >= 1.0 / max_period_m

    # Expose positive-frequency power for the plot (skip DC bin)
    extra = {"fft_freq_m_inv": freqs[1:], "fft_power": power[1:]}

    if not valid.any():
        return _make_failed("fft", line_length_m, n,
                            "No valid frequencies in the specified period range."), extra

    power_valid = np.where(valid, power, 0.0)
    dom_idx = int(np.argmax(power_valid))
    dom_freq = float(freqs[dom_idx])
    if dom_freq <= 0:
        return _make_failed("fft", line_length_m, n, "Dominant frequency is DC."), extra

    period_m = 1.0 / dom_freq
    freq_res = 1.0 / line_length_m if line_length_m > 0 else 0.0
    uncertainty_m = (period_m ** 2) * freq_res

    n_periods = line_length_m / period_m if period_m > 0 else 0.0
    mean_power = float(np.mean(power[valid]))
    snr = float(power[dom_idx]) / (mean_power + 1e-30)

    if n_periods < 2:
        quality, message = "weak", f"Short profile; ~{n_periods:.1f} periods sampled."
    elif snr > 10:
        quality = "good"
        message = f"Dominant FFT peak at {_fmt_m(period_m)} (SNR={snr:.0f})."
    elif snr > 3:
        quality = "weak"
        message = f"Moderate FFT peak at {_fmt_m(period_m)} (SNR={snr:.0f})."
    else:
        quality = "ambiguous"
        message = f"Weak FFT peak at {_fmt_m(period_m)} (SNR={snr:.1f}); no dominant period."

    return PeriodicityResult(
        method="fft",
        period_m=period_m,
        uncertainty_m=uncertainty_m,
        line_length_m=line_length_m,
        n_periods=n_periods,
        n_samples=n,
        quality=quality,
        message=message,
    ), extra


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_failed(method: str, line_length_m: float, n: int, message: str) -> PeriodicityResult:
    return PeriodicityResult(
        method=method,
        period_m=float("nan"),
        uncertainty_m=None,
        line_length_m=line_length_m,
        n_periods=0.0,
        n_samples=n,
        quality="failed",
        message=message,
    )


def _quality_from_n_periods(n_periods: float, prominence: float) -> tuple[str, str]:
    if n_periods >= 3 and prominence > 0.3:
        return "good", f"{n_periods:.1f} periods sampled; clear peak (prominence={prominence:.2f})."
    elif n_periods >= 1:
        return "weak", f"{n_periods:.1f} periods sampled; moderate peak (prominence={prominence:.2f})."
    else:
        return "failed", "Fewer than one period sampled."


def _fmt_m(period_m: float) -> str:
    if period_m < 1e-9:
        return f"{period_m * 1e10:.3g} Å"
    return f"{period_m * 1e9:.3g} nm"


# ── public formatting helpers (used by GUI) ───────────────────────────────────

def format_period(period_m: float) -> tuple[str, str]:
    """Return ``(value_str, unit_str)`` with sensible precision."""
    if period_m < 1e-9:
        return f"{period_m * 1e10:.3g}", "Å"
    return f"{period_m * 1e9:.3g}", "nm"


def format_result_text(
    result: PeriodicityResult,
    background: str = "",
    smoothing: str = "",
) -> str:
    """Return a plain-text summary of *result* suitable for copy-to-clipboard."""
    if math.isnan(result.period_m):
        period_line = "Period: N/A"
    else:
        val, unit = format_period(result.period_m)
        if result.uncertainty_m is not None and not math.isnan(result.uncertainty_m):
            scale = 1e10 if unit == "Å" else 1e9
            unc_val = result.uncertainty_m * scale
            period_line = f"Period: {val} ± {unc_val:.2g} {unit}"
        else:
            period_line = f"Period: {val} {unit}"

    length_nm = result.line_length_m * 1e9
    lines = [
        "Line periodicity",
        period_line,
        f"Line length: {length_nm:.3g} nm",
        f"Periods sampled: {result.n_periods:.1f}",
        f"Method: {result.method}",
    ]
    if background:
        lines.append(f"Background: {background}")
    if smoothing:
        lines.append(f"Smoothing: {smoothing}")
    return "\n".join(lines)
