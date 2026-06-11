"""Physics-correctness tests for the spectroscopy pipeline (2026-06-11 review).

Covers the seams the kernel-level tests miss: the VERT-header → setpoint
extraction that feeds setpoint normalization (the qPlus Δf-as-amps bug, fixed
on the scan path earlier but never on the spectroscopy path), time-axis
sanitisation, unit-label propagation through derivative/normalization, and
the documented pipeline order.
"""
from __future__ import annotations

import numpy as np
import pytest

from probeflow.io.spectroscopy import _createc_setpoint_a, _spec_freq_hz
from probeflow.spectroscopy.models import SpectrumDisplayOptions, SpectrumTrace
from probeflow.spectroscopy.transforms import make_displayed_spectrum


def _trace(y, x=None, *, x_unit="V", y_unit="A", y_channel="I",
           y_label="current", metadata=None) -> SpectrumTrace:
    y = np.asarray(y, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, y.size) if x is None else np.asarray(x, float)
    return SpectrumTrace(
        source_file="t.VERT", spectrum_id="s1",
        x_channel="V", y_channel=y_channel,
        x_raw=x, y_raw=y,
        x_label="Bias", y_label=y_label,
        x_unit=x_unit, y_unit=y_unit,
        metadata=dict(metadata or {}),
    )


# ── VERT setpoint extraction (qPlus Δf must never read as a current) ──────────

class TestVertSetpointExtraction:
    QPLUS_HDR = {
        "FBChannel": "4",   # PLL frequency-shift channel
        "PLLOn": "1",
        "SetPoint": "3.2",  # Δf in Hz — NOT a current
        "FBLogI": "700",    # stale leftover from the current loop
    }

    def test_qplus_header_yields_no_current_setpoint(self):
        """A constant-Δf qPlus header must not produce a setpoint in amps:
        before the fix this returned 3.2 (Hz read as A), and setpoint
        normalization silently rescaled spectra by ~10 orders of magnitude."""
        assert _createc_setpoint_a(self.QPLUS_HDR) is None

    def test_stm_current_feedback_reads_amps(self):
        hdr = {"FBChannel": "0", "PLLOn": "0", "Current[A]": "1e-10"}
        assert _createc_setpoint_a(hdr) == pytest.approx(1e-10)

    def test_stm_setpoint_key_reads_amps_when_feedback_is_current(self):
        hdr = {"PLLOn": "0", "SetPoint": "2.5e-10"}
        assert _createc_setpoint_a(hdr) == pytest.approx(2.5e-10)

    def test_fblogi_fallback_is_picoamps(self):
        hdr = {"FBChannel": "0", "FBLogI": "700"}
        assert _createc_setpoint_a(hdr) == pytest.approx(700e-12)

    def test_zero_and_negative_setpoints_are_unknown(self):
        assert _createc_setpoint_a({"Current[A]": "0"}) is None
        assert _createc_setpoint_a({"Current[A]": "-1e-10", "FBLogI": "0"}) is None

    def test_setpoint_normalization_blocks_instead_of_silently_rescaling(self):
        """End to end: with qPlus metadata the displayed-spectrum pipeline
        must refuse setpoint normalization (no finite setpoint), not divide
        by a frequency pretending to be a current."""
        trace = _trace(np.linspace(-1e-10, 1e-10, 32),
                       metadata={"setpoint_a": _createc_setpoint_a(self.QPLUS_HDR)})
        with pytest.raises(ValueError, match="setpoint"):
            make_displayed_spectrum(
                trace, SpectrumDisplayOptions(normalize_mode="setpoint"))


# ── Time-axis sanitisation ────────────────────────────────────────────────────

class TestSpecFreqSanitisation:
    def test_valid_frequency_passes_through(self):
        assert _spec_freq_hz({"SpecFreq": "512"}) == pytest.approx(512.0)

    def test_missing_uses_documented_default(self):
        assert _spec_freq_hz({}) == pytest.approx(1000.0)

    @pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf", "garbage"])
    def test_invalid_frequency_falls_back_instead_of_inf_time_axis(self, bad):
        """x = idx / spec_freq: a zero/negative/non-finite header value would
        turn the whole time axis into inf or negative seconds."""
        freq = _spec_freq_hz({"SpecFreq": bad})
        assert np.isfinite(freq) and freq > 0


# ── Unit-label propagation (invariant 7) ──────────────────────────────────────

class TestUnitPropagation:
    def test_derivative_updates_current_units_and_label(self):
        disp = make_displayed_spectrum(
            _trace(np.linspace(0, 1e-9, 32)),
            SpectrumDisplayOptions(derivative=True),
        )
        assert disp.y_unit == "A/V"
        assert disp.y_label == "numerical dI/dV"

    def test_derivative_of_unitless_y_reads_one_over_x_unit(self):
        disp = make_displayed_spectrum(
            _trace(np.linspace(0, 1, 32), y_unit="", y_channel="aux",
                   y_label="ratio"),
            SpectrumDisplayOptions(derivative=True),
        )
        assert disp.y_unit == "1/V"

    def test_derivative_over_unitless_x_keeps_y_unit(self):
        disp = make_displayed_spectrum(
            _trace(np.linspace(0, 1e-9, 32), x_unit=""),
            SpectrumDisplayOptions(derivative=True),
        )
        assert disp.y_unit == "A"

    def test_normalization_makes_units_relative(self):
        disp = make_displayed_spectrum(
            _trace(np.linspace(1e-12, 1e-9, 32)),
            SpectrumDisplayOptions(normalize_mode="max_abs"),
        )
        assert disp.y_unit == "relative"
        assert "divided by max |y|" in disp.y_label


# ── Pipeline order (invariant 2) ──────────────────────────────────────────────

class TestPipelineOrder:
    def test_normalization_applies_to_the_derivative_not_before_it(self):
        """max-abs normalization after the derivative must make the
        *derivative* peak 1. If normalization ran first, the derivative of a
        rescaled trace would carry an arbitrary 1/V scale instead."""
        x = np.linspace(-1.0, 1.0, 201)
        y = x ** 2  # dy/dx = 2x → max |dy/dx| = 2
        disp = make_displayed_spectrum(
            _trace(y, x),
            SpectrumDisplayOptions(derivative=True, normalize_mode="max_abs"),
        )
        assert float(np.nanmax(np.abs(disp.y_display))) == pytest.approx(1.0)

    def test_display_pipeline_metadata_records_documented_order(self):
        disp = make_displayed_spectrum(
            _trace(np.linspace(0, 1, 16)), SpectrumDisplayOptions())
        steps = disp.metadata["display_pipeline"]
        names = [s.split("=")[0] for s in steps[1:]]
        assert names == ["smoothing", "derivative", "normalization",
                         "outlier_mask", "vertical_offset"]

    def test_smoothing_runs_before_derivative(self):
        """Differentiation amplifies noise; the pipeline must smooth first.
        With heavy smoothing the derivative of noisy flat data stays small;
        if the derivative ran first the noise would already be amplified."""
        rng = np.random.default_rng(7)
        x = np.linspace(-1.0, 1.0, 401)
        y = 1e-10 * np.ones_like(x) + rng.normal(scale=1e-12, size=x.size)
        raw = make_displayed_spectrum(
            _trace(y, x), SpectrumDisplayOptions(derivative=True))
        smoothed = make_displayed_spectrum(
            _trace(y, x),
            SpectrumDisplayOptions(derivative=True, smoothing_mode="gaussian",
                                   smoothing_points=25),
        )
        assert (np.nanstd(smoothed.y_display)
                < 0.25 * np.nanstd(raw.y_display))

    def test_channel_normalization_with_derivative_is_rejected(self):
        trace = _trace(np.linspace(0, 1e-9, 16))
        with pytest.raises(ValueError, match="channel normalization"):
            make_displayed_spectrum(
                trace,
                SpectrumDisplayOptions(derivative=True,
                                       normalize_mode="channel",
                                       normalize_channel="I"),
                channel_lookup={"I": np.ones(16)},
            )

    def test_forward_backward_sweep_is_refused_not_differentiated(self):
        x = np.concatenate([np.linspace(-1, 1, 16), np.linspace(1, -1, 16)])
        y = np.linspace(0, 1e-9, 32)
        with pytest.raises(ValueError, match="monotonic"):
            make_displayed_spectrum(
                _trace(y, x), SpectrumDisplayOptions(derivative=True))
