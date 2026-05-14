"""Seam tests for the measurement module.

These tests exercise multi-step workflows across module boundaries:
importer / viewer / ROI / measurement functions / table / export.
Each test is named after the seam it covers.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from probeflow.core.roi import ROI
from probeflow.measurements.export import (
    measurements_to_csv_text,
    measurements_to_json_text,
    measurements_to_tsv,
)
from probeflow.measurements.features import detect_local_maxima, feature_maxima_result
from probeflow.measurements.fft_points import (
    fft_from_point_mask,
    point_fft_summary_result,
    points_to_mask,
)
from probeflow.measurements.image import (
    line_profile_measurement,
    roi_statistics,
    step_height_from_rois,
)
from probeflow.measurements.models import MeasurementResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _rect_roi(x, y, w, h, name="roi"):
    return ROI.new("rectangle", {"x": x, "y": y, "width": w, "height": h}, name=name)


def _gaussian_image(peaks, shape=(64, 64), sigma=1.5):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.zeros(shape, dtype=float)
    for x0, y0, amp in peaks:
        image += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return image


# ── seam 1: measurement stores ROI identity; survives ROI deletion ────────────

def test_roi_deletion_seam_result_remains_interpretable():
    """Measurement result must not depend on a live ROI object after creation."""
    image = np.arange(16, dtype=float).reshape(4, 4)
    roi = _rect_roi(1, 1, 2, 2, name="terrace")

    result = roi_statistics(
        image,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="nm",
    )

    # Simulate deletion: lose the reference entirely
    roi_id_at_measurement = result.context["roi_id"]
    roi_name_at_measurement = result.context["roi_name"]
    del roi  # noqa: F821 — intentional deletion to simulate UI removal

    assert result.context["roi_id"] == roi_id_at_measurement
    assert result.context["roi_name"] == roi_name_at_measurement
    assert result.values["mean_height"] == pytest.approx(7.5)


# ── seam 2: processing state — data_basis is recorded in context ──────────────

def test_processing_state_seam_data_basis_recorded():
    """Result must record whether data were measured on raw or processed image."""
    raw = np.zeros((8, 8), dtype=float)
    processed = raw - raw.mean()  # trivial plane subtraction
    roi = _rect_roi(2, 2, 4, 4)

    raw_result = roi_statistics(
        raw,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="pm",
        data_basis="raw_image",
    )
    proc_result = roi_statistics(
        processed,
        measurement_id="M0002",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="pm",
        data_basis="processed_image",
    )

    assert raw_result.context["data_basis"] == "raw_image"
    assert proc_result.context["data_basis"] == "processed_image"
    assert raw_result.context["data_basis"] != proc_result.context["data_basis"]


# ── seam 3: display state — measurement values are from data, not display ──────

def test_display_state_seam_values_independent_of_display_scaling():
    """Measurement functions receive the data array directly.

    This test simulates a display-scaled copy of the image and verifies that
    passing the original data array gives data-correct values, not display values.
    """
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    display_scaled = data * 255.0 / data.max()  # crude display normalisation
    roi = _rect_roi(0, 0, 2, 2)

    data_result = roi_statistics(
        data,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="nm",
    )
    display_result = roi_statistics(
        display_scaled,
        measurement_id="M0002",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        height_unit="nm",
    )

    assert data_result.values["mean_height"] == pytest.approx(2.5)
    assert display_result.values["mean_height"] == pytest.approx(2.5 * 255.0 / 4.0)
    assert data_result.values["mean_height"] != display_result.values["mean_height"]


# ── seam 4: channel seam — distinct channel labels in results ─────────────────

def test_channel_seam_forward_backward_labels_are_distinct():
    """Two measurements on the same image with different channel labels must differ."""
    image = np.ones((4, 4), dtype=float)
    roi = _rect_roi(0, 0, 4, 4)

    fwd = roi_statistics(
        image,
        measurement_id="M0001",
        source_label="scan:Z_fwd",
        channel="Z_fwd",
        roi=roi,
        height_unit="pm",
    )
    bwd = roi_statistics(
        image,
        measurement_id="M0002",
        source_label="scan:Z_bwd",
        channel="Z_bwd",
        roi=roi,
        height_unit="pm",
    )

    assert fwd.channel != bwd.channel
    assert fwd.source_label != bwd.source_label


# ── seam 5: unit seam — units propagate through all export formats ────────────

def test_unit_seam_units_survive_csv_and_json_export():
    """Physical units must be present in every supported export format."""
    image = np.full((8, 8), 5.0)
    roi = _rect_roi(1, 1, 6, 6)

    stats = roi_statistics(
        image,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        roi=roi,
        x_unit="nm",
        y_unit="nm",
        height_unit="pm",
    )
    distance = np.linspace(0, 10, 50)
    profile = np.sin(distance)
    line = line_profile_measurement(
        distance,
        profile,
        measurement_id="M0002",
        source_label="scan:Z",
        channel="Z",
        x_unit="nm",
        y_unit="pm",
    )

    results = [stats, line]

    csv_text = measurements_to_csv_text(results)
    tsv_text = measurements_to_tsv(results)
    payload = json.loads(measurements_to_json_text(results))

    assert "nm" in csv_text
    assert "pm" in csv_text
    assert "nm" in tsv_text
    assert "pm" in tsv_text
    m0 = payload["measurements"][0]
    assert m0["x_unit"] == "nm"
    assert m0["z_unit"] == "pm"


# ── seam 6: copy/export seam — non-finite values handled predictably ──────────

def test_copy_export_seam_non_finite_values_serialise_without_crash():
    """Non-finite floats must not raise during TSV/CSV/JSON export."""
    result = MeasurementResult(
        measurement_id="M0001",
        kind="roi_stats",
        source_label="scan:Z",
        source_path=None,
        channel="Z",
        x_unit="nm",
        y_unit="nm",
        z_unit="pm",
        values={
            "mean_height": float("nan"),
            "rms_roughness": float("inf"),
            "n_finite_pixels": 0,
        },
        context={"data_basis": "processed_image"},
    )

    tsv = measurements_to_tsv([result])
    csv_text = measurements_to_csv_text([result])
    json_text = measurements_to_json_text([result])
    payload = json.loads(json_text)

    m = payload["measurements"][0]
    assert m["values"]["mean_height"] == "nan"
    assert m["values"]["rms_roughness"] == "inf"
    assert "nan" in tsv or "nan" in csv_text


# ── seam 7: JSON export includes provenance fields ────────────────────────────

def test_json_export_includes_provenance_fields():
    """JSON export must carry schema_version, created_at, and probeflow_version."""
    result = MeasurementResult(
        measurement_id="M0001",
        kind="roi_stats",
        source_label="scan:Z",
        source_path="/path/to/scan.sxm",
        channel="Z",
        x_unit="nm",
        y_unit="nm",
        z_unit="pm",
        values={"mean_height": 1.0},
        context={"data_basis": "processed_image"},
    )

    payload = json.loads(measurements_to_json_text([result]))

    assert payload["export_type"] == "probeflow_measurements"
    assert payload["schema_version"] == "1"
    assert "created_at" in payload
    assert "T" in payload["created_at"]  # ISO-8601 shape
    assert "probeflow_version" in payload


# ── seam 8: step height stores both ROI identities ───────────────────────────

def test_step_height_seam_records_both_roi_identities():
    """Step-height result must record ROI IDs/names for both ROIs at measurement time."""
    image = np.zeros((8, 8), dtype=float)
    image[:, 4:] = 3.0
    lower = _rect_roi(0, 0, 4, 8, name="lower_terrace")
    upper = _rect_roi(4, 0, 4, 8, name="upper_terrace")

    result = step_height_from_rois(
        image,
        lower,
        upper,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        height_unit="pm",
    )

    assert result.values["height_difference"] == pytest.approx(3.0)
    assert result.context["roi_a_name"] == "lower_terrace"
    assert result.context["roi_b_name"] == "upper_terrace"
    assert result.context["roi_a_id"] is not None
    assert result.context["roi_b_id"] is not None
    assert result.z_unit == "pm"


# ── seam 9: feature maxima → point mask → FFT end-to-end ─────────────────────

def test_feature_maxima_to_fft_seam_end_to_end():
    """Detect points, create mask, compute FFT, collect result — no crashes."""
    image = _gaussian_image([(16, 16, 5.0), (48, 48, 4.0)])
    roi = _rect_roi(0, 0, 64, 64, name="scan_area")
    roi_mask = roi.to_mask((64, 64))

    points = detect_local_maxima(
        image,
        threshold_mode="percentile",
        threshold_value=90.0,
        min_distance_px=5,
        roi_mask=roi_mask,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        channel="Z",
        source_label="scan:Z",
        roi_id=roi.id,
    )
    assert len(points) >= 2

    maxima_result = feature_maxima_result(
        points,
        measurement_id="M0001",
        source_label="scan:Z",
        channel="Z",
        threshold_mode="percentile",
        threshold_value=90.0,
        min_distance_px=5,
        roi_id=roi.id,
    )

    mask = points_to_mask(points, (64, 64), radius_px=1)
    fft_result = fft_from_point_mask(
        mask,
        pixel_size_x=0.5,
        pixel_size_y=0.5,
        spatial_unit="nm",
        n_points=len(points),
        radius_px=1,
    )
    fft_summary = point_fft_summary_result(
        fft_result,
        measurement_id="M0002",
        source_label="scan:Z",
        channel="Z",
    )

    assert maxima_result.values["n_points"] == len(points)
    assert maxima_result.context["roi_id"] == roi.id
    assert fft_summary.kind == "point_fft"
    assert fft_summary.values["n_points"] == len(points)
    assert fft_result.units == "cycles/nm"

    json_text = measurements_to_json_text([maxima_result, fft_summary])
    payload = json.loads(json_text)
    assert len(payload["measurements"]) == 2


# ── seam 10: error handling — invalid inputs raise, not silently produce rows ──

def test_error_handling_seam_empty_roi_raises_not_returns_bad_result():
    """roi_statistics on an all-NaN ROI must raise ValueError, not return a result."""
    image = np.full((8, 8), np.nan)
    roi = _rect_roi(0, 0, 8, 8)

    with pytest.raises(ValueError, match="finite"):
        roi_statistics(
            image,
            measurement_id="M0001",
            source_label="scan:Z",
            channel="Z",
            roi=roi,
        )


def test_error_handling_seam_step_height_one_empty_roi_raises():
    """step_height_from_rois with an all-NaN ROI must raise, not return garbage."""
    image = np.zeros((8, 8), dtype=float)
    image[:4, :] = np.nan
    lower = _rect_roi(0, 0, 8, 4, name="nan_region")
    upper = _rect_roi(0, 4, 8, 4, name="valid_region")

    with pytest.raises(ValueError, match="finite"):
        step_height_from_rois(
            image,
            lower,
            upper,
            measurement_id="M0001",
            source_label="scan:Z",
            channel="Z",
        )
