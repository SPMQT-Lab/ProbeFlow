"""Tests for generic measurement models and export helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.measurements.export import (
    measurement_to_flat_dict,
    measurements_to_json_text,
    measurements_to_tsv,
)
from probeflow.measurements.adapters import legacy_measurement_to_result
from probeflow.measurements.image import (
    line_profile_measurement,
    roi_statistics,
    step_height_from_rois,
)
from probeflow.measurements.models import MeasurementResult, measurement_main_value
from probeflow.measurements.spectrum import spectrum_delta_to_result
from probeflow.spectroscopy.measurement import measure_delta, nearest_finite_point
from probeflow.spectroscopy.models import DisplayedSpectrum, SpectrumDisplayOptions


def test_measurement_export_flattens_values_and_context():
    result = MeasurementResult(
        measurement_id="M0001",
        kind="spectrum_delta",
        source_label="spec:I",
        source_path="/tmp/spec.VERT",
        channel="I",
        x_unit="V",
        y_unit="pA",
        values={"dx": 0.25, "dy": 6.4},
        context={"data_basis": "displayed_trace"},
    )

    flat = measurement_to_flat_dict(result)
    text = measurements_to_tsv([result])
    payload = json.loads(measurements_to_json_text([result]))

    assert flat["value.dx"] == 0.25
    assert flat["context.data_basis"] == "displayed_trace"
    assert "value.dx" in text
    assert payload["measurements"][0]["values"]["dy"] == 6.4
    assert measurement_main_value(result) == ("dy", 6.4, "pA")


def test_legacy_measurement_adapter_preserves_context_and_units():
    from probeflow.analysis.measurements import MeasurementResult as LegacyResult

    legacy = LegacyResult(
        id="M?",
        kind="roi_stats",
        source="scan:Height",
        channel="Height",
        roi_id="roi-1",
        summary="mean = 2.5 nm",
        values={"mean": 2.5, "n_pixels": 4},
        units={"mean": "nm"},
        context={"source_path": "/tmp/scan.sxm", "roi_name": "terrace"},
    )

    result = legacy_measurement_to_result(legacy, "M0009")

    assert result.measurement_id == "M0009"
    assert result.kind == "roi_stats"
    assert result.source_path == "/tmp/scan.sxm"
    assert result.values["mean_height"] == pytest.approx(2.5)
    assert result.z_unit == "nm"
    assert result.context["roi_id"] == "roi-1"
    assert result.context["roi_name"] == "terrace"


def test_spectrum_delta_converts_to_generic_measurement_with_display_context():
    trace = DisplayedSpectrum(
        source_file="/tmp/spec.VERT",
        spectrum_id="spec",
        label="spec I",
        x_channel="Bias",
        y_channel="I",
        x_display=np.array([0.0, 1.0, 2.0]),
        y_display=np.array([0.0, 2.0, 4.0]),
        mask=None,
        options=SpectrumDisplayOptions(
            smoothing_mode="savgol",
            smoothing_points=7,
            normalize_mode="max_abs",
            vertical_offset=0.5,
        ),
        metadata={},
        x_label="Bias",
        y_label="Current",
        x_unit="V",
        y_unit="relative",
    )
    p1 = nearest_finite_point(trace, 0.0, 0.0, max_normalized_distance=None)
    p2 = nearest_finite_point(trace, 2.0, 4.0, max_normalized_distance=None)

    result = spectrum_delta_to_result(
        measure_delta(p1, p2),
        measurement_id="M0002",
        trace=trace,
    )

    assert result.kind == "spectrum_delta"
    assert result.values["dx"] == pytest.approx(2.0)
    assert result.context["data_basis"] == "displayed_trace"
    assert result.context["normalization"] == "max_abs"
    assert result.context["smoothing_window"] == 7


def test_roi_statistics_use_masked_processed_values_without_mutation():
    image = np.arange(16, dtype=float).reshape(4, 4)
    before = image.copy()
    roi = ROI.new("rectangle", {"x": 1, "y": 1, "width": 2, "height": 2}, name="terrace")

    result = roi_statistics(
        image,
        measurement_id="M0003",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        pixel_size_x=2.0,
        pixel_size_y=3.0,
        x_unit="nm",
        y_unit="nm",
        height_unit="nm",
    )

    np.testing.assert_array_equal(image, before)
    assert result.values["n_finite_pixels"] == 4
    assert result.values["mean_height"] == pytest.approx(7.5)
    assert result.values["area"] == pytest.approx(24.0)
    assert result.context["roi_name"] == "terrace"


def test_step_height_from_two_rois_reports_difference():
    image = np.zeros((4, 4), dtype=float)
    image[:, 2:] = 2.5
    left = ROI.new("rectangle", {"x": 0, "y": 0, "width": 2, "height": 4}, name="lower")
    right = ROI.new("rectangle", {"x": 2, "y": 0, "width": 2, "height": 4}, name="upper")

    result = step_height_from_rois(
        image,
        left,
        right,
        measurement_id="M0004",
        source_label="scan:Z",
        channel="Z",
        height_unit="nm",
    )

    assert result.kind == "step_height"
    assert result.values["height_difference"] == pytest.approx(2.5)
    assert result.values["roi_a_n"] == 8
    assert result.context["roi_b_name"] == "upper"


def test_line_profile_measurement_summarizes_profile():
    distance = np.array([0.0, 1.0, 2.0])
    profile = np.array([2.0, 4.0, 8.0])

    result = line_profile_measurement(
        distance,
        profile,
        measurement_id="M0005",
        source_label="scan:Z",
        channel="Z",
        x_unit="nm",
        y_unit="nm",
        p0=(0.0, 0.0),
        p1=(2.0, 0.0),
        roi_id="roi-line",
        roi_name="line",
    )

    assert result.kind == "line_profile"
    assert result.values["length"] == pytest.approx(2.0)
    assert result.values["height_difference"] == pytest.approx(6.0)
    assert "height_peak_to_peak" not in result.values
    assert "height_mean" not in result.values
    assert result.values["x2"] == pytest.approx(2.0)
    assert result.context["roi_id"] == "roi-line"
