"""Tests for non-destructive spectroscopy display transforms and exports."""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.spectroscopy.export import (
    displayed_spectra_to_csv_text,
    displayed_spectra_to_json_text,
)
from probeflow.spectroscopy.models import SpectrumDisplayOptions, SpectrumTrace
from probeflow.spectroscopy.transforms import (
    apply_outlier_mask,
    apply_smoothing,
    make_displayed_spectrum,
)


def _trace(y=None) -> SpectrumTrace:
    x = np.linspace(-1.0, 1.0, 9)
    if y is None:
        y = x + 2.0
    return SpectrumTrace(
        source_file="spec.dat",
        spectrum_id="spec",
        x_channel="Bias",
        y_channel="Current",
        x_raw=x,
        y_raw=np.asarray(y, dtype=float),
        x_label="Bias (V)",
        y_label="Current",
        x_unit="V",
        y_unit="A",
    )


def test_smoothing_does_not_mutate_raw_input():
    y = np.linspace(0.0, 1.0, 9)
    before = y.copy()

    out = apply_smoothing(y, mode="gaussian", points=5)

    np.testing.assert_array_equal(y, before)
    assert out.shape == y.shape


def test_savgol_requires_odd_window():
    with pytest.raises(ValueError, match="odd"):
        apply_smoothing(np.arange(8, dtype=float), mode="savgol", points=4, polyorder=2)


def test_mad_outlier_mask_detects_single_spike():
    x = np.arange(7, dtype=float)
    y = np.array([1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0])

    x_out, y_out, keep = apply_outlier_mask(x, y, mode="mad", threshold=6.0)

    assert keep.tolist() == [True, True, True, False, True, True, True]
    assert len(x_out) == 6
    assert 20.0 not in y_out


def test_jump_outlier_mask_detects_single_spike():
    x = np.arange(5, dtype=float)
    y = np.array([1.0, 1.0, 20.0, 1.0, 1.0])

    _x_out, y_out, keep = apply_outlier_mask(x, y, mode="jump", threshold=6.0)

    assert keep.tolist() == [True, True, False, True, True]
    assert 20.0 not in y_out


def test_disabling_outlier_mask_keeps_all_finite_points():
    x = np.arange(3, dtype=float)
    y = np.array([1.0, 50.0, 1.0])

    _, y_out, keep = apply_outlier_mask(x, y, mode="none")

    assert keep.tolist() == [True, True, True]
    np.testing.assert_allclose(y_out, y)


def test_constant_normalization_and_vertical_offset():
    trace = _trace(np.array([2.0, 4.0, 6.0]))
    trace = SpectrumTrace(
        **{**trace.__dict__, "x_raw": np.array([0.0, 1.0, 2.0])}
    )
    opts = SpectrumDisplayOptions(
        normalize_mode="constant",
        normalize_constant=2.0,
        vertical_offset=-1.0,
    )

    displayed = make_displayed_spectrum(trace, opts)

    np.testing.assert_allclose(displayed.y_display, [0.0, 1.0, 2.0])
    assert displayed.options.vertical_offset == -1.0


def test_channel_normalization_uses_matching_denominator():
    trace = _trace(np.array([2.0, 4.0, 8.0]))
    trace = SpectrumTrace(
        **{**trace.__dict__, "x_raw": np.array([0.0, 1.0, 2.0])}
    )
    opts = SpectrumDisplayOptions(normalize_mode="channel", normalize_channel="Setpoint")

    displayed = make_displayed_spectrum(
        trace,
        opts,
        channel_lookup={"Setpoint": np.array([2.0, 2.0, 4.0])},
    )

    np.testing.assert_allclose(displayed.y_display, [1.0, 2.0, 2.0])


def test_didv_trace_preserves_length_and_units():
    x = np.linspace(-1.0, 1.0, 101)
    y = 2e-9 * x + 1e-9
    displayed = make_displayed_spectrum(
        SpectrumTrace(
            source_file="spec.dat",
            spectrum_id="spec",
            x_channel="Bias",
            y_channel="I",
            x_raw=x,
            y_raw=y,
            x_label="Bias (V)",
            y_label="Current channel",
            x_unit="V",
            y_unit="A",
        ),
        SpectrumDisplayOptions(derivative=True),
    )

    assert displayed.y_display.shape == x.shape
    assert displayed.y_unit == "A/V"
    assert displayed.y_label == "dI/dV"
    np.testing.assert_allclose(displayed.y_display[5:-5], 2e-9, rtol=1e-10)


def test_derivative_rejects_channel_normalization():
    with pytest.raises(ValueError, match="channel normalization"):
        make_displayed_spectrum(
            _trace(),
            SpectrumDisplayOptions(
                derivative=True,
                normalize_mode="channel",
                normalize_channel="denominator",
            ),
            channel_lookup={"denominator": np.ones(9)},
        )


def test_export_csv_and_json_values_match_displayed_arrays():
    displayed = make_displayed_spectrum(
        _trace(np.arange(9, dtype=float)),
        SpectrumDisplayOptions(outlier_mode="mad", outlier_threshold=6.0),
    )

    csv_text = displayed_spectra_to_csv_text([displayed])
    json_payload = json.loads(displayed_spectra_to_json_text([displayed]))

    assert "source_file,spectrum_id,trace_label" in csv_text
    assert ",spec,spec Current," in csv_text
    assert json_payload["traces"][0]["y"] == [float(v) for v in displayed.y_display]
    assert json_payload["traces"][0]["excluded_indices"] == displayed.excluded_indices


def test_csv_omits_masked_column_and_records_excluded_indices():
    displayed = make_displayed_spectrum(
        _trace(np.array([1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
        SpectrumDisplayOptions(outlier_mode="mad", outlier_threshold=6.0),
    )

    csv_text = displayed_spectra_to_csv_text([displayed])

    assert ",masked" not in csv_text
    assert "# trace_0_excluded_indices,[2]" in csv_text
