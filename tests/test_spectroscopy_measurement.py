"""Tests for display-trace spectroscopy measurement helpers."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.spectroscopy.measurement import (
    format_measurement_summary,
    measure_delta,
    measurement_to_tsv,
    nearest_finite_point,
    nearest_point_across_traces,
)
from probeflow.spectroscopy.models import DisplayedSpectrum, SpectrumDisplayOptions


def _displayed(
    *,
    label: str = "spec I",
    source_file: str = "file.VERT",
    y_channel: str = "I",
    x=None,
    y=None,
    mask=None,
    y_unit: str = "pA",
) -> DisplayedSpectrum:
    x_arr = np.asarray([0.0, 1.0, 2.0, 3.0] if x is None else x, dtype=float)
    y_arr = np.asarray([0.0, 2.0, 4.0, 6.0] if y is None else y, dtype=float)
    return DisplayedSpectrum(
        source_file=source_file,
        spectrum_id=source_file.rsplit(".", 1)[0],
        label=label,
        x_channel="Bias (V)",
        y_channel=y_channel,
        x_display=x_arr,
        y_display=y_arr,
        mask=mask,
        options=SpectrumDisplayOptions(),
        metadata={},
        x_label="Bias",
        y_label="Current",
        x_unit="V",
        y_unit=y_unit,
    )


def test_nearest_finite_point_ignores_nan_values():
    trace = _displayed(y=[0.0, np.nan, 4.0, 6.0])

    point = nearest_finite_point(trace, 0.9, 1.9, max_normalized_distance=None)

    assert point is not None
    assert point.index == 0
    assert point.trace_name == "file.VERT:I"


def test_nearest_finite_point_ignores_display_mask_when_aligned():
    trace = _displayed(mask=np.array([True, False, True, True]))

    point = nearest_finite_point(trace, 1.0, 2.0, max_normalized_distance=None)

    assert point is not None
    assert point.index != 1


def test_measure_delta_reports_dx_dy_slope_and_units():
    trace = _displayed()
    p1 = nearest_finite_point(trace, 1.0, 2.0, max_normalized_distance=None)
    p2 = nearest_finite_point(trace, 3.0, 6.0, max_normalized_distance=None)

    measurement = measure_delta(p1, p2)

    assert measurement.dx == pytest.approx(2.0)
    assert measurement.dy == pytest.approx(4.0)
    assert measurement.slope == pytest.approx(2.0)
    assert measurement.slope_unit == "pA/V"
    summary = format_measurement_summary(measurement)
    assert "ΔV" in summary
    assert "ΔI" in summary
    assert "slope" in summary
    assert "dx" in measurement_to_tsv(measurement)


def test_measure_delta_rejects_points_from_different_traces():
    p1 = nearest_finite_point(_displayed(source_file="a.VERT"), 0.0, 0.0)
    p2 = nearest_finite_point(_displayed(source_file="b.VERT"), 0.0, 0.0)

    with pytest.raises(ValueError, match="same trace"):
        measure_delta(p1, p2)


def test_nearest_point_across_traces_selects_nearest_trace():
    far = _displayed(label="far", source_file="far.VERT", y=[100.0, 101.0, 102.0, 103.0])
    near = _displayed(label="near", source_file="near.VERT", y=[0.0, 1.0, 2.0, 3.0])

    point = nearest_point_across_traces([far, near], 2.0, 2.1, max_normalized_distance=0.2)

    assert point is not None
    assert point.source_file == "near.VERT"
    assert point.index == 2


def test_nearest_point_returns_none_when_click_is_not_close():
    trace = _displayed()

    point = nearest_finite_point(trace, 100.0, 100.0, max_normalized_distance=0.01)

    assert point is None
